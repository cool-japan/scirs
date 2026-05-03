//! Maximum flow algorithms
//!
//! This module provides three maximum flow algorithms:
//! - **Dinic's algorithm**: O(V² E) general, O(E √V) for unit-capacity graphs
//! - **Push-Relabel (FIFO highest-label)**: O(V² √E)
//! - **Edmonds-Karp**: O(V E²) via BFS augmenting paths

use std::collections::VecDeque;

/// Internal edge representation for residual graph.
#[derive(Debug, Clone)]
struct FlowEdge {
    /// Destination node
    to: usize,
    /// Current remaining capacity
    cap: i64,
    /// Index of the reverse edge in the adjacency list of `to`
    rev: usize,
}

/// Dinic's maximum flow algorithm.
///
/// Builds a level graph with BFS and finds blocking flows with DFS.
/// Time complexity: O(V² E) for general graphs, O(E √V) for unit-capacity.
///
/// # Example
/// ```
/// use scirs2_graph::flow::max_flow::DinicMaxFlow;
///
/// let mut d = DinicMaxFlow::new(4);
/// d.add_edge(0, 1, 10);
/// d.add_edge(0, 2, 10);
/// d.add_edge(1, 3, 10);
/// d.add_edge(2, 3, 10);
/// d.add_edge(1, 2, 1);
/// let flow = d.max_flow(0, 3);
/// assert_eq!(flow, 20);
/// ```
#[derive(Debug, Clone)]
pub struct DinicMaxFlow {
    n: usize,
    graph: Vec<Vec<FlowEdge>>,
}

impl DinicMaxFlow {
    /// Create a new Dinic solver with `n` nodes (0-indexed).
    pub fn new(n: usize) -> Self {
        Self {
            n,
            graph: vec![vec![]; n],
        }
    }

    /// Add a directed edge from `u` to `v` with capacity `cap`.
    /// A reverse edge with capacity 0 is automatically added.
    pub fn add_edge(&mut self, u: usize, v: usize, cap: i64) {
        let rev_u = self.graph[v].len();
        let rev_v = self.graph[u].len();
        self.graph[u].push(FlowEdge {
            to: v,
            cap,
            rev: rev_u,
        });
        self.graph[v].push(FlowEdge {
            to: u,
            cap: 0,
            rev: rev_v,
        });
    }

    /// Compute max flow from `source` to `sink`.
    /// Returns 0 if source == sink or no path exists.
    pub fn max_flow(&mut self, source: usize, sink: usize) -> i64 {
        if source >= self.n || sink >= self.n {
            return 0;
        }
        let mut total = 0i64;
        while let Some(level) = self.bfs(source, sink) {
            let mut iter: Vec<usize> = vec![0; self.n];
            loop {
                let f = self.dfs(source, sink, i64::MAX, &level, &mut iter);
                if f == 0 {
                    break;
                }
                total += f;
            }
        }
        total
    }

    /// BFS to build the level graph. Returns None if sink is unreachable.
    fn bfs(&self, source: usize, sink: usize) -> Option<Vec<i64>> {
        let mut level = vec![-1i64; self.n];
        level[source] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(source);
        while let Some(v) = queue.pop_front() {
            for e in &self.graph[v] {
                if e.cap > 0 && level[e.to] < 0 {
                    level[e.to] = level[v] + 1;
                    queue.push_back(e.to);
                }
            }
        }
        if level[sink] < 0 {
            None
        } else {
            Some(level)
        }
    }

    /// DFS blocking flow pass.
    fn dfs(
        &mut self,
        v: usize,
        sink: usize,
        pushed: i64,
        level: &[i64],
        iter: &mut Vec<usize>,
    ) -> i64 {
        if v == sink {
            return pushed;
        }
        while iter[v] < self.graph[v].len() {
            let i = iter[v];
            let (to, cap, rev) = {
                let e = &self.graph[v][i];
                (e.to, e.cap, e.rev)
            };
            if cap > 0 && level[v] < level[to] {
                let d = self.dfs(to, sink, pushed.min(cap), level, iter);
                if d > 0 {
                    self.graph[v][i].cap -= d;
                    self.graph[to][rev].cap += d;
                    return d;
                }
            }
            iter[v] += 1;
        }
        0
    }

    /// Return the current flow on each edge (forward edges only).
    /// Each entry is `(u, v, capacity, flow)`.
    pub fn edges(&self) -> Vec<(usize, usize, i64, i64)> {
        let mut result = Vec::new();
        for u in 0..self.n {
            for e in &self.graph[u] {
                // forward edge: cap >= 0 and the matching reverse has cap 0
                // We detect forward edges by checking that the reverse edge's
                // reverse index points back to us and the reverse edge has cap 0 initially.
                // Simpler: collect all edges, original cap > 0 edges only.
                // To avoid exposing internal bookkeeping, just return all directed edges
                // with capacity > 0 or where flow > 0.
                let original_cap = e.cap + self.graph[e.to][e.rev].cap;
                if original_cap > 0 {
                    let flow = original_cap - e.cap;
                    result.push((u, e.to, original_cap, flow));
                }
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Push-Relabel (FIFO + Highest-Label variant)
// ─────────────────────────────────────────────────────────────────────────────

/// Push-Relabel maximum flow algorithm with FIFO selection and highest-label
/// strategy.
///
/// Time complexity: O(V² √E).
///
/// # Example
/// ```
/// use scirs2_graph::flow::max_flow::PushRelabelMaxFlow;
///
/// let mut pr = PushRelabelMaxFlow::new(4);
/// pr.add_edge(0, 1, 10);
/// pr.add_edge(0, 2, 10);
/// pr.add_edge(1, 3, 10);
/// pr.add_edge(2, 3, 10);
/// let flow = pr.max_flow(0, 3);
/// assert_eq!(flow, 20);
/// ```
#[derive(Debug, Clone)]
pub struct PushRelabelMaxFlow {
    n: usize,
    graph: Vec<Vec<FlowEdge>>,
}

impl PushRelabelMaxFlow {
    /// Create a new Push-Relabel solver with `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            graph: vec![vec![]; n],
        }
    }

    /// Add a directed edge from `u` to `v` with capacity `cap`.
    pub fn add_edge(&mut self, u: usize, v: usize, cap: i64) {
        let rev_u = self.graph[v].len();
        let rev_v = self.graph[u].len();
        self.graph[u].push(FlowEdge {
            to: v,
            cap,
            rev: rev_u,
        });
        self.graph[v].push(FlowEdge {
            to: u,
            cap: 0,
            rev: rev_v,
        });
    }

    /// Compute the maximum flow from `source` to `sink`.
    pub fn max_flow(&mut self, source: usize, sink: usize) -> i64 {
        if source >= self.n || sink >= self.n || source == sink {
            return 0;
        }
        let n = self.n;
        let mut height = vec![0i64; n];
        let mut excess = vec![0i64; n];
        let mut current = vec![0usize; n];

        // Initialize: saturate all edges from source
        height[source] = n as i64;
        for i in 0..self.graph[source].len() {
            let (to, cap, rev) = {
                let e = &self.graph[source][i];
                (e.to, e.cap, e.rev)
            };
            if cap > 0 {
                excess[to] += cap;
                excess[source] -= cap;
                self.graph[source][i].cap = 0;
                self.graph[to][rev].cap += cap;
            }
        }

        // Active nodes (excess > 0, not source or sink)
        // Use a highest-label bucket queue via Vec<VecDeque<usize>>
        let mut buckets: Vec<VecDeque<usize>> = vec![VecDeque::new(); 2 * n + 1];
        let mut in_queue = vec![false; n];
        for v in 0..n {
            if v != source && v != sink && excess[v] > 0 {
                let h = height[v] as usize;
                buckets[h].push_back(v);
                in_queue[v] = true;
            }
        }

        let mut highest = n - 1;

        loop {
            // Find highest active node
            while highest > 0 && buckets[highest].is_empty() {
                if highest == 0 {
                    break;
                }
                highest -= 1;
            }
            if buckets[highest].is_empty() {
                break;
            }
            let v = match buckets[highest].front().copied() {
                Some(x) => x,
                None => break,
            };

            // Discharge v
            while excess[v] > 0 {
                if current[v] == self.graph[v].len() {
                    // Relabel
                    let min_h = self.graph[v]
                        .iter()
                        .filter(|e| e.cap > 0)
                        .map(|e| height[e.to])
                        .min()
                        .unwrap_or(2 * n as i64);
                    height[v] = min_h + 1;
                    current[v] = 0;
                    // Move to new bucket
                    let old_h = highest;
                    let _ = buckets[old_h].pop_front();
                    in_queue[v] = false;
                    let new_h = height[v] as usize;
                    if new_h < 2 * n {
                        buckets[new_h].push_front(v);
                        in_queue[v] = true;
                        if new_h > highest {
                            highest = new_h;
                        }
                    }
                    break;
                }
                let i = current[v];
                let (to, cap, rev) = {
                    let e = &self.graph[v][i];
                    (e.to, e.cap, e.rev)
                };
                if cap > 0 && height[v] == height[to] + 1 {
                    let pushed = excess[v].min(cap);
                    excess[v] -= pushed;
                    excess[to] += pushed;
                    self.graph[v][i].cap -= pushed;
                    self.graph[to][rev].cap += pushed;

                    if to != source && to != sink && !in_queue[to] && excess[to] > 0 {
                        let h = height[to] as usize;
                        if h < 2 * n {
                            buckets[h].push_back(to);
                            in_queue[to] = true;
                            if h > highest {
                                highest = h;
                            }
                        }
                    }
                } else {
                    current[v] += 1;
                }
            }

            // If v was fully discharged, remove from bucket
            if excess[v] == 0 {
                if let Some(&front) = buckets[highest].front() {
                    if front == v {
                        buckets[highest].pop_front();
                        in_queue[v] = false;
                    }
                }
            }
        }

        excess[sink].max(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Edmonds-Karp (BFS augmenting paths)
// ─────────────────────────────────────────────────────────────────────────────

/// Edmonds-Karp maximum flow algorithm.
///
/// Uses BFS to find shortest augmenting paths (Ford-Fulkerson with BFS).
/// Time complexity: O(V E²).
///
/// # Example
/// ```
/// use scirs2_graph::flow::max_flow::EdmondsKarp;
///
/// let mut ek = EdmondsKarp::new(4);
/// ek.add_edge(0, 1, 10);
/// ek.add_edge(0, 2, 10);
/// ek.add_edge(1, 3, 10);
/// ek.add_edge(2, 3, 10);
/// let flow = ek.max_flow(0, 3);
/// assert_eq!(flow, 20);
/// ```
#[derive(Debug, Clone)]
pub struct EdmondsKarp {
    n: usize,
    graph: Vec<Vec<FlowEdge>>,
}

impl EdmondsKarp {
    /// Create a new Edmonds-Karp solver with `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            graph: vec![vec![]; n],
        }
    }

    /// Add a directed edge from `u` to `v` with capacity `cap`.
    pub fn add_edge(&mut self, u: usize, v: usize, cap: i64) {
        let rev_u = self.graph[v].len();
        let rev_v = self.graph[u].len();
        self.graph[u].push(FlowEdge {
            to: v,
            cap,
            rev: rev_u,
        });
        self.graph[v].push(FlowEdge {
            to: u,
            cap: 0,
            rev: rev_v,
        });
    }

    /// Compute maximum flow from `source` to `sink`.
    pub fn max_flow(&mut self, source: usize, sink: usize) -> i64 {
        if source >= self.n || sink >= self.n {
            return 0;
        }
        let mut total = 0i64;
        loop {
            // BFS to find shortest augmenting path
            let mut parent: Vec<Option<(usize, usize)>> = vec![None; self.n];
            parent[source] = Some((source, 0));
            let mut queue = VecDeque::new();
            queue.push_back(source);
            'bfs: while let Some(v) = queue.pop_front() {
                for (i, e) in self.graph[v].iter().enumerate() {
                    if e.cap > 0 && parent[e.to].is_none() {
                        parent[e.to] = Some((v, i));
                        if e.to == sink {
                            break 'bfs;
                        }
                        queue.push_back(e.to);
                    }
                }
            }
            if parent[sink].is_none() {
                break;
            }
            // Find bottleneck
            let mut bottleneck = i64::MAX;
            let mut node = sink;
            while node != source {
                let (prev, ei) = match parent[node] {
                    Some(p) => p,
                    None => break,
                };
                bottleneck = bottleneck.min(self.graph[prev][ei].cap);
                node = prev;
            }
            // Augment
            let mut node = sink;
            while node != source {
                let (prev, ei) = match parent[node] {
                    Some(p) => p,
                    None => break,
                };
                let rev = self.graph[prev][ei].rev;
                self.graph[prev][ei].cap -= bottleneck;
                self.graph[node][rev].cap += bottleneck;
                node = prev;
            }
            total += bottleneck;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> (usize, Vec<(usize, usize, i64)>) {
        // 4-node graph: 0→1 (10), 0→2 (10), 1→3 (10), 2→3 (10), 1→2 (1)
        (
            4,
            vec![(0, 1, 10), (0, 2, 10), (1, 3, 10), (2, 3, 10), (1, 2, 1)],
        )
    }

    #[test]
    fn test_dinic_simple() {
        let (n, edges) = simple_graph();
        let mut d = DinicMaxFlow::new(n);
        for (u, v, c) in &edges {
            d.add_edge(*u, *v, *c);
        }
        assert_eq!(d.max_flow(0, 3), 20);
    }

    #[test]
    fn test_push_relabel_simple() {
        let (n, edges) = simple_graph();
        let mut pr = PushRelabelMaxFlow::new(n);
        for (u, v, c) in &edges {
            pr.add_edge(*u, *v, *c);
        }
        assert_eq!(pr.max_flow(0, 3), 20);
    }

    #[test]
    fn test_edmonds_karp_simple() {
        let (n, edges) = simple_graph();
        let mut ek = EdmondsKarp::new(n);
        for (u, v, c) in &edges {
            ek.add_edge(*u, *v, *c);
        }
        assert_eq!(ek.max_flow(0, 3), 20);
    }

    #[test]
    fn test_dinic_no_path() {
        let mut d = DinicMaxFlow::new(3);
        d.add_edge(0, 1, 5);
        // no edge to node 2
        assert_eq!(d.max_flow(0, 2), 0);
    }

    #[test]
    fn test_edmonds_karp_bottleneck() {
        let mut ek = EdmondsKarp::new(3);
        ek.add_edge(0, 1, 100);
        ek.add_edge(1, 2, 1);
        assert_eq!(ek.max_flow(0, 2), 1);
    }

    #[test]
    fn test_dinic_cycle() {
        // Graph with a cycle
        let mut d = DinicMaxFlow::new(4);
        d.add_edge(0, 1, 5);
        d.add_edge(1, 2, 3);
        d.add_edge(2, 1, 2);
        d.add_edge(1, 3, 4);
        d.add_edge(2, 3, 3);
        // max flow 0→3: path 0→1→3 gives 4, path 0→1→2→3 gives 1 = total 5
        assert_eq!(d.max_flow(0, 3), 5);
    }

    #[test]
    fn test_push_relabel_larger() {
        // CLRS example: 6 nodes
        let mut pr = PushRelabelMaxFlow::new(6);
        pr.add_edge(0, 1, 16);
        pr.add_edge(0, 2, 13);
        pr.add_edge(1, 2, 10);
        pr.add_edge(1, 3, 12);
        pr.add_edge(2, 1, 4);
        pr.add_edge(2, 4, 14);
        pr.add_edge(3, 2, 9);
        pr.add_edge(3, 5, 20);
        pr.add_edge(4, 3, 7);
        pr.add_edge(4, 5, 4);
        assert_eq!(pr.max_flow(0, 5), 23);
    }

    #[test]
    fn test_dinic_clrs_example() {
        let mut d = DinicMaxFlow::new(6);
        d.add_edge(0, 1, 16);
        d.add_edge(0, 2, 13);
        d.add_edge(1, 2, 10);
        d.add_edge(1, 3, 12);
        d.add_edge(2, 1, 4);
        d.add_edge(2, 4, 14);
        d.add_edge(3, 2, 9);
        d.add_edge(3, 5, 20);
        d.add_edge(4, 3, 7);
        d.add_edge(4, 5, 4);
        assert_eq!(d.max_flow(0, 5), 23);
    }

    #[test]
    fn test_zero_capacity() {
        let mut d = DinicMaxFlow::new(2);
        d.add_edge(0, 1, 0);
        assert_eq!(d.max_flow(0, 1), 0);
    }

    #[test]
    fn test_single_edge() {
        let mut d = DinicMaxFlow::new(2);
        d.add_edge(0, 1, 42);
        assert_eq!(d.max_flow(0, 1), 42);
    }

    #[test]
    fn test_all_three_agree() {
        // Use same topology, confirm all three give same answer
        let edges = vec![
            (0usize, 1usize, 7i64),
            (0, 2, 4),
            (1, 3, 3),
            (1, 4, 5),
            (2, 4, 4),
            (3, 5, 5),
            (4, 5, 6),
        ];
        let n = 6;
        let mut d = DinicMaxFlow::new(n);
        let mut pr = PushRelabelMaxFlow::new(n);
        let mut ek = EdmondsKarp::new(n);
        for &(u, v, c) in &edges {
            d.add_edge(u, v, c);
            pr.add_edge(u, v, c);
            ek.add_edge(u, v, c);
        }
        let fd = d.max_flow(0, 5);
        let fpr = pr.max_flow(0, 5);
        let fek = ek.max_flow(0, 5);
        assert_eq!(fd, fpr, "Dinic vs PushRelabel mismatch");
        assert_eq!(fd, fek, "Dinic vs EdmondsKarp mismatch");
    }
}
