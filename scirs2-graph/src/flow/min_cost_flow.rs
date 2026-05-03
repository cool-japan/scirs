//! Minimum cost flow algorithms
//!
//! Provides two implementations:
//! - [`MinCostFlow`]: Successive shortest paths via SPFA (Bellman-Ford with queue)
//! - [`PotentialMinCostFlow`]: Johnson's potentials + Dijkstra (faster for non-negative costs)

use std::collections::VecDeque;

/// Internal edge for the min-cost-flow residual graph.
#[derive(Debug, Clone)]
struct McfEdge {
    to: usize,
    cap: i64,
    cost: i64,
    rev: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// MinCostFlow — SPFA successive shortest paths
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum cost flow using successive shortest paths with SPFA (Bellman-Ford
/// with a deque).
///
/// Supports negative costs; detects negative-cost cycles.
///
/// # Example
/// ```
/// use scirs2_graph::flow::min_cost_flow::MinCostFlow;
///
/// let mut mcf = MinCostFlow::new(4);
/// mcf.add_edge(0, 1, 3, 1);
/// mcf.add_edge(0, 2, 2, 5);
/// mcf.add_edge(1, 2, 1, 1);
/// mcf.add_edge(1, 3, 2, 3);
/// mcf.add_edge(2, 3, 2, 2);
/// let (flow, cost) = mcf.min_cost_max_flow(0, 3);
/// assert_eq!(flow, 4);
/// ```
#[derive(Debug, Clone)]
pub struct MinCostFlow {
    n: usize,
    graph: Vec<Vec<McfEdge>>,
}

impl MinCostFlow {
    /// Create a new MinCostFlow solver with `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            graph: vec![vec![]; n],
        }
    }

    /// Add a directed edge `u → v` with capacity `cap` and cost `cost` per unit.
    /// Negative costs are supported.
    pub fn add_edge(&mut self, u: usize, v: usize, cap: i64, cost: i64) {
        let rev_u = self.graph[v].len();
        let rev_v = self.graph[u].len();
        self.graph[u].push(McfEdge {
            to: v,
            cap,
            cost,
            rev: rev_u,
        });
        self.graph[v].push(McfEdge {
            to: u,
            cap: 0,
            cost: -cost,
            rev: rev_v,
        });
    }

    /// Send at most `max_flow` units from `source` to `sink` with minimum cost.
    ///
    /// Returns `(actual_flow, total_cost)`.
    pub fn min_cost_flow(&mut self, source: usize, sink: usize, max_flow: i64) -> (i64, i64) {
        let mut flow = 0i64;
        let mut cost = 0i64;
        while flow < max_flow {
            let (dist, prev_node, prev_edge) = match self.spfa(source, sink) {
                Some(x) => x,
                None => break,
            };
            // Bottleneck along the path
            let mut d = max_flow - flow;
            let mut v = sink;
            while v != source {
                let pn = prev_node[v];
                let pe = prev_edge[v];
                d = d.min(self.graph[pn][pe].cap);
                v = pn;
            }
            // Augment
            let mut v = sink;
            while v != source {
                let pn = prev_node[v];
                let pe = prev_edge[v];
                let rev = self.graph[pn][pe].rev;
                self.graph[pn][pe].cap -= d;
                self.graph[v][rev].cap += d;
                v = pn;
            }
            flow += d;
            cost += d * dist[sink];
        }
        (flow, cost)
    }

    /// Compute minimum cost maximum flow from `source` to `sink`.
    ///
    /// Returns `(total_flow, total_cost)`.
    pub fn min_cost_max_flow(&mut self, source: usize, sink: usize) -> (i64, i64) {
        self.min_cost_flow(source, sink, i64::MAX)
    }

    /// SPFA (Shortest Path Faster Algorithm) to find min-cost augmenting path.
    /// Returns (dist, prev_node, prev_edge) or None if sink is unreachable.
    fn spfa(&self, source: usize, sink: usize) -> Option<(Vec<i64>, Vec<usize>, Vec<usize>)> {
        const INF: i64 = i64::MAX / 2;
        let n = self.n;
        let mut dist = vec![INF; n];
        let mut in_queue = vec![false; n];
        let mut prev_node = vec![0usize; n];
        let mut prev_edge = vec![0usize; n];
        let mut count = vec![0usize; n]; // for negative-cycle detection

        dist[source] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(source);
        in_queue[source] = true;

        while let Some(v) = queue.pop_front() {
            in_queue[v] = false;
            for (i, e) in self.graph[v].iter().enumerate() {
                if e.cap > 0 && dist[v] < INF && dist[v] + e.cost < dist[e.to] {
                    dist[e.to] = dist[v] + e.cost;
                    prev_node[e.to] = v;
                    prev_edge[e.to] = i;
                    if !in_queue[e.to] {
                        in_queue[e.to] = true;
                        count[e.to] += 1;
                        if count[e.to] >= n {
                            // Negative cycle detected — cannot route more flow
                            return None;
                        }
                        // SLF (Shortest Label First) optimization
                        if let Some(&front) = queue.front() {
                            if dist[e.to] < dist[front] {
                                queue.push_front(e.to);
                            } else {
                                queue.push_back(e.to);
                            }
                        } else {
                            queue.push_back(e.to);
                        }
                    }
                }
            }
        }
        if dist[sink] == INF {
            None
        } else {
            Some((dist, prev_node, prev_edge))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PotentialMinCostFlow — Johnson's potentials + Dijkstra
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum cost flow using Johnson's potentials with Dijkstra.
///
/// More efficient than SPFA for graphs with non-negative costs.
/// Uses initial Bellman-Ford to compute potentials, then maintains them
/// across successive Dijkstra calls to handle residual negative edges.
///
/// # Example
/// ```
/// use scirs2_graph::flow::min_cost_flow::PotentialMinCostFlow;
///
/// let mut mcf = PotentialMinCostFlow::new(4);
/// mcf.add_edge(0, 1, 3, 1);
/// mcf.add_edge(0, 2, 2, 5);
/// mcf.add_edge(1, 2, 1, 1);
/// mcf.add_edge(1, 3, 2, 3);
/// mcf.add_edge(2, 3, 2, 2);
/// let (flow, cost) = mcf.min_cost_max_flow(0, 3);
/// assert_eq!(flow, 4);
/// ```
#[derive(Debug, Clone)]
pub struct PotentialMinCostFlow {
    n: usize,
    graph: Vec<Vec<McfEdge>>,
}

impl PotentialMinCostFlow {
    /// Create a new PotentialMinCostFlow solver with `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            graph: vec![vec![]; n],
        }
    }

    /// Add a directed edge `u → v` with capacity `cap` and cost `cost`.
    /// Costs should be non-negative for best performance; negative-cost cycles
    /// are not supported.
    pub fn add_edge(&mut self, u: usize, v: usize, cap: i64, cost: i64) {
        let rev_u = self.graph[v].len();
        let rev_v = self.graph[u].len();
        self.graph[u].push(McfEdge {
            to: v,
            cap,
            cost,
            rev: rev_u,
        });
        self.graph[v].push(McfEdge {
            to: u,
            cap: 0,
            cost: -cost,
            rev: rev_v,
        });
    }

    /// Compute minimum cost maximum flow from `source` to `sink`.
    ///
    /// Returns `(total_flow, total_cost)`.
    pub fn min_cost_max_flow(&mut self, source: usize, sink: usize) -> (i64, i64) {
        self.min_cost_flow(source, sink, i64::MAX)
    }

    /// Send at most `max_flow` units with minimum cost.
    ///
    /// Returns `(actual_flow, total_cost)`.
    pub fn min_cost_flow(&mut self, source: usize, sink: usize, max_flow: i64) -> (i64, i64) {
        // Initialize potentials via Bellman-Ford from source
        let mut potential = self.bellman_ford(source);

        let mut flow = 0i64;
        let mut cost = 0i64;

        while flow < max_flow {
            let (dist, prev_node, prev_edge) = match self.dijkstra(source, sink, &potential) {
                Some(x) => x,
                None => break,
            };

            // Update potentials
            for v in 0..self.n {
                if dist[v] < i64::MAX / 2 {
                    potential[v] += dist[v];
                }
            }

            // Bottleneck
            let mut d = max_flow - flow;
            let mut v = sink;
            while v != source {
                let pn = prev_node[v];
                let pe = prev_edge[v];
                d = d.min(self.graph[pn][pe].cap);
                v = pn;
            }

            // Augment
            let mut v = sink;
            while v != source {
                let pn = prev_node[v];
                let pe = prev_edge[v];
                let rev = self.graph[pn][pe].rev;
                self.graph[pn][pe].cap -= d;
                self.graph[v][rev].cap += d;
                v = pn;
            }

            flow += d;
            cost += d * potential[sink];
        }
        (flow, cost)
    }

    /// Bellman-Ford from `source` to compute initial potentials.
    fn bellman_ford(&self, source: usize) -> Vec<i64> {
        const INF: i64 = i64::MAX / 2;
        let n = self.n;
        let mut dist = vec![INF; n];
        dist[source] = 0;
        for _ in 0..n - 1 {
            let mut updated = false;
            for v in 0..n {
                if dist[v] == INF {
                    continue;
                }
                for e in &self.graph[v] {
                    if e.cap > 0 && dist[v] + e.cost < dist[e.to] {
                        dist[e.to] = dist[v] + e.cost;
                        updated = true;
                    }
                }
            }
            if !updated {
                break;
            }
        }
        // Clamp unreachable nodes to 0 for potential use
        for d in &mut dist {
            if *d == INF {
                *d = 0;
            }
        }
        dist
    }

    /// Dijkstra with Johnson's potentials (reduced costs ≥ 0).
    fn dijkstra(
        &self,
        source: usize,
        sink: usize,
        potential: &[i64],
    ) -> Option<(Vec<i64>, Vec<usize>, Vec<usize>)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        const INF: i64 = i64::MAX / 2;
        let n = self.n;
        let mut dist = vec![INF; n];
        let mut prev_node = vec![0usize; n];
        let mut prev_edge = vec![0usize; n];
        dist[source] = 0;

        // (dist, node)
        let mut heap = BinaryHeap::new();
        heap.push(Reverse((0i64, source)));

        while let Some(Reverse((d, v))) = heap.pop() {
            if d > dist[v] {
                continue;
            }
            for (i, e) in self.graph[v].iter().enumerate() {
                if e.cap > 0 {
                    // Reduced cost: cost + potential[v] - potential[e.to]
                    let reduced = e.cost + potential[v] - potential[e.to];
                    let nd = dist[v] + reduced;
                    if nd < dist[e.to] {
                        dist[e.to] = nd;
                        prev_node[e.to] = v;
                        prev_edge[e.to] = i;
                        heap.push(Reverse((nd, e.to)));
                    }
                }
            }
        }

        if dist[sink] == INF {
            None
        } else {
            Some((dist, prev_node, prev_edge))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the small 4-node test graph.
    fn build_mcf_graph<F: FnMut(usize, usize, i64, i64)>(mut add: F) {
        add(0, 1, 3, 1);
        add(0, 2, 2, 5);
        add(1, 2, 1, 1);
        add(1, 3, 2, 3);
        add(2, 3, 2, 2);
    }

    #[test]
    fn test_spfa_min_cost_max_flow() {
        let mut mcf = MinCostFlow::new(4);
        build_mcf_graph(|u, v, c, w| mcf.add_edge(u, v, c, w));
        let (flow, cost) = mcf.min_cost_max_flow(0, 3);
        assert_eq!(flow, 4);
        // Expected cost: 3 units via 0→1→3 (cost 4 each) + 1 unit via 0→2→3 (cost 7) = 19
        assert_eq!(cost, 19);
    }

    #[test]
    fn test_potential_min_cost_max_flow() {
        let mut mcf = PotentialMinCostFlow::new(4);
        build_mcf_graph(|u, v, c, w| mcf.add_edge(u, v, c, w));
        let (flow, cost) = mcf.min_cost_max_flow(0, 3);
        assert_eq!(flow, 4);
        assert_eq!(cost, 19);
    }

    #[test]
    fn test_partial_flow() {
        let mut mcf = MinCostFlow::new(4);
        build_mcf_graph(|u, v, c, w| mcf.add_edge(u, v, c, w));
        let (flow, cost) = mcf.min_cost_flow(0, 3, 2);
        assert_eq!(flow, 2);
        // 2 units via cheapest path 0→1→3 cost 4 each = 8
        assert_eq!(cost, 8);
    }

    #[test]
    fn test_negative_cost_edge() {
        // Graph with a negative-cost edge (but no negative cycle)
        let mut mcf = MinCostFlow::new(4);
        mcf.add_edge(0, 1, 5, 2);
        mcf.add_edge(1, 2, 5, -1); // negative cost edge
        mcf.add_edge(2, 3, 5, 3);
        let (flow, cost) = mcf.min_cost_max_flow(0, 3);
        assert_eq!(flow, 5);
        assert_eq!(cost, (2 - 1 + 3) * 5); // 20
    }

    #[test]
    fn test_no_path() {
        let mut mcf = MinCostFlow::new(4);
        mcf.add_edge(0, 1, 5, 1);
        // No edge to reach node 3
        let (flow, _cost) = mcf.min_cost_max_flow(0, 3);
        assert_eq!(flow, 0);
    }

    #[test]
    fn test_spfa_and_potential_agree() {
        let edges = vec![
            (0usize, 1usize, 4i64, 2i64),
            (0, 2, 2, 5),
            (1, 2, 1, 1),
            (1, 3, 3, 3),
            (2, 4, 3, 2),
            (3, 4, 2, 4),
            (3, 5, 2, 1),
            (4, 5, 4, 2),
        ];
        let n = 6;
        let mut spfa = MinCostFlow::new(n);
        let mut pot = PotentialMinCostFlow::new(n);
        for &(u, v, c, w) in &edges {
            spfa.add_edge(u, v, c, w);
            pot.add_edge(u, v, c, w);
        }
        let (f1, c1) = spfa.min_cost_max_flow(0, 5);
        let (f2, c2) = pot.min_cost_max_flow(0, 5);
        assert_eq!(f1, f2, "flow mismatch between SPFA and Potential");
        assert_eq!(c1, c2, "cost mismatch between SPFA and Potential");
    }

    #[test]
    fn test_single_edge() {
        let mut mcf = MinCostFlow::new(2);
        mcf.add_edge(0, 1, 7, 3);
        let (flow, cost) = mcf.min_cost_max_flow(0, 1);
        assert_eq!(flow, 7);
        assert_eq!(cost, 21);
    }

    #[test]
    fn test_zero_capacity() {
        let mut mcf = MinCostFlow::new(2);
        mcf.add_edge(0, 1, 0, 5);
        let (flow, cost) = mcf.min_cost_max_flow(0, 1);
        assert_eq!(flow, 0);
        assert_eq!(cost, 0);
    }

    #[test]
    fn test_parallel_edges() {
        // Two parallel edges with different costs
        let mut mcf = MinCostFlow::new(2);
        mcf.add_edge(0, 1, 3, 5);
        mcf.add_edge(0, 1, 2, 2);
        let (flow, cost) = mcf.min_cost_max_flow(0, 1);
        assert_eq!(flow, 5);
        // Cheapest 2 units via cost-2 edge, then 3 units via cost-5 edge
        assert_eq!(cost, 2 * 2 + 3 * 5); // 19
    }
}
