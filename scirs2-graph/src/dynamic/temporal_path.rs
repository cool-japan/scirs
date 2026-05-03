//! Temporal path finding: earliest arrival, latest departure, and fastest paths.
//!
//! Edges are represented as `(src, dst, departure_time, travel_time)` tuples so
//! that the departure from a node can be no earlier than the current arrival time
//! at that node (i.e. *time-respecting* paths).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A temporal path: a sequence of time-stamped hops through a graph.
#[derive(Debug, Clone)]
pub struct TemporalPath {
    /// Ordered list of node identifiers visited.
    pub nodes: Vec<usize>,
    /// Arrival time at each node.
    pub times: Vec<f64>,
    /// Departure time of each traversed edge.
    pub edge_times: Vec<f64>,
    /// Elapsed time between source departure and final arrival.
    pub total_duration: f64,
    /// Number of hops.
    pub n_hops: usize,
}

impl TemporalPath {
    /// Create a one-node path starting at `src` at `start_time`.
    pub fn new(src: usize, start_time: f64) -> Self {
        Self {
            nodes: vec![src],
            times: vec![start_time],
            edge_times: Vec::new(),
            total_duration: 0.0,
            n_hops: 0,
        }
    }

    /// Extend the path by one hop.
    pub fn extend(&mut self, node: usize, edge_time: f64, arrival_time: f64) {
        self.nodes.push(node);
        self.times.push(arrival_time);
        self.edge_times.push(edge_time);
        let start = self.times.first().copied().unwrap_or(0.0);
        self.total_duration = arrival_time - start;
        self.n_hops += 1;
    }

    /// Final arrival time of this path.
    pub fn arrival_time(&self) -> f64 {
        self.times.last().copied().unwrap_or(0.0)
    }
}

/// Fixed-point scale factor used when converting `f64` times to `i64` for the
/// binary heap.  Values are rounded to nanosecond precision.
const TIME_SCALE: f64 = 1_000_000_000.0;

/// Dijkstra-style algorithms adapted for temporal (time-respecting) paths.
pub struct TemporalDijkstra;

impl TemporalDijkstra {
    /// Find the **earliest-arrival** path from `src` to `dst` in a temporal graph.
    ///
    /// # Arguments
    /// * `src`        – source node id
    /// * `dst`        – destination node id
    /// * `start_time` – earliest time the source may be departed
    /// * `edges`      – temporal edge list: `(u, v, departure_time, travel_time)`
    /// * `n_nodes`    – total number of nodes (node ids must be in `0..n_nodes`)
    ///
    /// Returns `None` when no time-respecting path exists.
    pub fn earliest_arrival(
        src: usize,
        dst: usize,
        start_time: f64,
        edges: &[(usize, usize, f64, f64)],
        n_nodes: usize,
    ) -> Option<TemporalPath> {
        let mut arrival = vec![f64::INFINITY; n_nodes];
        arrival[src] = start_time;

        // Build per-node adjacency lists sorted by departure time.
        let mut adj: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); n_nodes];
        for &(u, v, dep, travel) in edges {
            adj[u].push((v, dep, travel));
        }
        for list in &mut adj {
            list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Min-heap: (arrival_time_scaled, node)
        let mut pq: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
        pq.push(Reverse(((start_time * TIME_SCALE) as i64, src)));

        // predecessor[v] = (predecessor_node, edge_departure_time)
        let mut prev: Vec<Option<(usize, f64)>> = vec![None; n_nodes];

        while let Some(Reverse((t_scaled, u))) = pq.pop() {
            let t = t_scaled as f64 / TIME_SCALE;
            if t > arrival[u] {
                continue;
            }
            if u == dst {
                break;
            }

            // Walk temporal edges departing at or after current arrival.
            for &(v, dep, travel) in &adj[u] {
                if dep >= t {
                    let arrive = dep + travel;
                    if arrive < arrival[v] {
                        arrival[v] = arrive;
                        prev[v] = Some((u, dep));
                        pq.push(Reverse(((arrive * TIME_SCALE) as i64, v)));
                    }
                }
            }
        }

        if arrival[dst].is_infinite() {
            return None;
        }

        // Reconstruct node sequence by walking predecessors backwards.
        let mut rev_nodes = vec![dst];
        let mut cursor = dst;
        while let Some((p, _)) = prev[cursor] {
            rev_nodes.push(p);
            cursor = p;
            if cursor == src {
                break;
            }
        }
        rev_nodes.reverse();

        // Build TemporalPath by replaying the chosen edges.
        let mut path = TemporalPath::new(src, start_time);
        let mut curr_arr = start_time;

        for window in rev_nodes.windows(2) {
            let (u, v) = (window[0], window[1]);
            // Find the earliest-departing edge from u to v that we can catch.
            let chosen = adj[u]
                .iter()
                .find(|&&(nv, dep, _)| nv == v && dep >= curr_arr);
            if let Some(&(_, dep, travel)) = chosen {
                path.extend(v, dep, dep + travel);
                curr_arr = dep + travel;
            }
        }

        Some(path)
    }

    /// Find the **latest-departure** path that still arrives at `dst` by `deadline`.
    ///
    /// Uses a time-reversed earliest-arrival search on a reversed graph.
    pub fn latest_departure(
        src: usize,
        dst: usize,
        deadline: f64,
        edges: &[(usize, usize, f64, f64)],
        n_nodes: usize,
    ) -> Option<TemporalPath> {
        // Reverse the graph: each edge (u→v, dep, travel) becomes (v→u, arr, travel)
        // where arr = dep + travel is used as the "reversed departure" time, and we
        // maximise this time (i.e. minimise negative time).
        let reversed: Vec<(usize, usize, f64, f64)> = edges
            .iter()
            .map(|&(u, v, dep, travel)| (v, u, dep + travel, travel))
            .collect();

        // Run earliest-arrival on reversed graph from dst, treating the deadline as
        // a "start time" in the reversed timeline.  We want paths where we depart
        // as late as possible, which corresponds to the reversed departure time
        // being as large as possible.  We invert time by using `deadline - time`.
        let inverted: Vec<(usize, usize, f64, f64)> = reversed
            .iter()
            .map(|&(u, v, arr, travel)| (u, v, deadline - arr, travel))
            .collect();

        let path = Self::earliest_arrival(dst, src, 0.0, &inverted, n_nodes)?;

        // Convert inverted times back to real times.
        let real_times: Vec<f64> = path.times.iter().map(|&t| deadline - t).collect();
        let real_edges: Vec<f64> = path.edge_times.iter().map(|&t| deadline - t).collect();

        let mut result = TemporalPath::new(src, real_times.last().copied().unwrap_or(0.0));
        // Nodes are src-order in original graph but we reconstructed from dst; reverse.
        let mut rev_nodes = path.nodes.clone();
        rev_nodes.reverse();
        let mut rev_times = real_times;
        rev_times.reverse();
        let mut rev_edges = real_edges;
        rev_edges.reverse();

        result.nodes = rev_nodes;
        result.times = rev_times;
        result.edge_times = rev_edges;
        result.n_hops = path.n_hops;
        result.total_duration = path.total_duration;

        Some(result)
    }

    /// Find the **fastest** temporal path (minimises total elapsed duration).
    ///
    /// The search extends the state space with the departure time so that we
    /// track `(node, departure_time)` pairs and minimise `arrival - start_time`.
    pub fn fastest_path(
        src: usize,
        dst: usize,
        start_time: f64,
        edges: &[(usize, usize, f64, f64)],
        n_nodes: usize,
    ) -> Option<TemporalPath> {
        // State: (duration_scaled, node, actual_arrival_time_scaled)
        // We minimise duration = arrival_time - start_time.
        let mut adj: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); n_nodes];
        for &(u, v, dep, travel) in edges {
            adj[u].push((v, dep, travel));
        }
        for list in &mut adj {
            list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // best_duration[node] = shortest duration found so far to reach node
        let mut best_dur = vec![f64::INFINITY; n_nodes];
        best_dur[src] = 0.0;

        // heap: (duration_scaled, node, arrival_time_scaled)
        let mut pq: BinaryHeap<Reverse<(i64, usize, i64)>> = BinaryHeap::new();
        pq.push(Reverse((0, src, (start_time * TIME_SCALE) as i64)));

        let mut prev: Vec<Option<(usize, f64)>> = vec![None; n_nodes];

        while let Some(Reverse((dur_scaled, u, arr_scaled))) = pq.pop() {
            let dur = dur_scaled as f64 / TIME_SCALE;
            let arr = arr_scaled as f64 / TIME_SCALE;
            if dur > best_dur[u] {
                continue;
            }
            if u == dst {
                break;
            }

            for &(v, dep, travel) in &adj[u] {
                if dep >= arr {
                    let new_arr = dep + travel;
                    let new_dur = new_arr - start_time;
                    if new_dur < best_dur[v] {
                        best_dur[v] = new_dur;
                        prev[v] = Some((u, dep));
                        pq.push(Reverse((
                            (new_dur * TIME_SCALE) as i64,
                            v,
                            (new_arr * TIME_SCALE) as i64,
                        )));
                    }
                }
            }
        }

        if best_dur[dst].is_infinite() {
            return None;
        }

        // Reconstruct path.
        let mut rev_nodes = vec![dst];
        let mut cursor = dst;
        while let Some((p, _)) = prev[cursor] {
            rev_nodes.push(p);
            cursor = p;
            if cursor == src {
                break;
            }
        }
        rev_nodes.reverse();

        let mut path = TemporalPath::new(src, start_time);
        let mut curr_arr = start_time;
        for window in rev_nodes.windows(2) {
            let (u, v) = (window[0], window[1]);
            let chosen = adj[u]
                .iter()
                .find(|&&(nv, dep, _)| nv == v && dep >= curr_arr);
            if let Some(&(_, dep, travel)) = chosen {
                path.extend(v, dep, dep + travel);
                curr_arr = dep + travel;
            }
        }

        Some(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small temporal graph:
    ///
    /// 0→1 at t=1 (travel 1)
    /// 1→2 at t=3 (travel 1)
    /// 0→2 at t=5 (travel 1)   ← direct but later
    fn small_edges() -> Vec<(usize, usize, f64, f64)> {
        vec![(0, 1, 1.0, 1.0), (1, 2, 3.0, 1.0), (0, 2, 5.0, 1.0)]
    }

    #[test]
    fn test_earliest_arrival_two_hop() {
        let edges = small_edges();
        let path =
            TemporalDijkstra::earliest_arrival(0, 2, 0.0, &edges, 3).expect("path should exist");
        // Via 0→1 (arr=2) then 1→2 (arr=4) is earlier than 0→2 direct (arr=6)
        assert_eq!(path.nodes, vec![0, 1, 2], "nodes={:?}", path.nodes);
        assert!((path.arrival_time() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_earliest_arrival_direct_faster() {
        // 0→2 direct at t=1 (arr=2) vs 0→1 at t=1 (arr=2) then 1→2 at t=5 (arr=6)
        let edges = vec![(0, 1, 1.0, 1.0), (1, 2, 5.0, 1.0), (0, 2, 1.0, 1.0)];
        let path =
            TemporalDijkstra::earliest_arrival(0, 2, 0.0, &edges, 3).expect("path should exist");
        assert_eq!(path.nodes, vec![0, 2], "expected direct path");
        assert!((path.arrival_time() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_no_path_returns_none() {
        // Edge only goes 0→1; destination is 2 with no incoming edge.
        let edges = vec![(0, 1, 1.0, 1.0)];
        let result = TemporalDijkstra::earliest_arrival(0, 2, 0.0, &edges, 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_fastest_path() {
        let edges = small_edges();
        let path = TemporalDijkstra::fastest_path(0, 2, 0.0, &edges, 3).expect("path should exist");
        // Duration via 0→1→2 = 4.0; via direct 0→2 = 6.0
        assert!(
            path.total_duration <= 4.0 + 1e-9,
            "duration={}",
            path.total_duration
        );
    }
}
