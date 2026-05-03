//! Link streams: continuous-time temporal graph model (Latapy et al. 2018).
//!
//! A *link stream* L = (T, V, E) consists of:
//! - a time interval T = [α, ω],
//! - a set of nodes V,
//! - a set of temporal edges E ⊆ T × V × V, each with an associated duration.

use std::collections::HashSet;

/// A temporal edge in a link stream: active during `[start, end]`.
#[derive(Debug, Clone)]
pub struct TemporalEdge {
    /// Source node.
    pub src: usize,
    /// Destination node.
    pub dst: usize,
    /// Activation start time.
    pub start: f64,
    /// Activation end time.
    pub end: f64,
}

impl TemporalEdge {
    /// Construct a new temporal edge.
    pub fn new(src: usize, dst: usize, start: f64, end: f64) -> Self {
        Self {
            src,
            dst,
            start,
            end,
        }
    }

    /// Duration of this contact.
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }

    /// Whether the edge is active at time `t`.
    pub fn is_active_at(&self, t: f64) -> bool {
        t >= self.start && t <= self.end
    }
}

/// A continuous-time temporal network represented as a link stream.
pub struct LinkStream {
    /// All temporal edges in the stream.
    pub edges: Vec<TemporalEdge>,
    /// All node identifiers present in the stream.
    pub nodes: Vec<usize>,
    /// Start of the observation window.
    pub time_start: f64,
    /// End of the observation window.
    pub time_end: f64,
}

impl LinkStream {
    /// Create an empty link stream over the interval `[time_start, time_end]`.
    pub fn new(time_start: f64, time_end: f64) -> Self {
        Self {
            edges: Vec::new(),
            nodes: Vec::new(),
            time_start,
            time_end,
        }
    }

    /// Add a temporal edge; both endpoints are registered as nodes if absent.
    pub fn add_edge(&mut self, src: usize, dst: usize, start: f64, end: f64) {
        for &n in &[src, dst] {
            if !self.nodes.contains(&n) {
                self.nodes.push(n);
            }
        }
        self.edges.push(TemporalEdge::new(src, dst, start, end));
    }

    /// Instantaneous degree of node `u` at time `t`.
    pub fn degree_at(&self, u: usize, t: f64) -> usize {
        self.edges
            .iter()
            .filter(|e| (e.src == u || e.dst == u) && e.is_active_at(t))
            .count()
    }

    /// Time-averaged degree of node `u`, estimated by uniform sampling.
    ///
    /// `n_samples` controls accuracy; higher values reduce discretisation error.
    pub fn average_degree(&self, u: usize, n_samples: usize) -> f64 {
        if n_samples == 0 {
            return 0.0;
        }
        let dt = (self.time_end - self.time_start) / n_samples as f64;
        let sum: f64 = (0..n_samples)
            .map(|i| {
                let t = self.time_start + (i as f64 + 0.5) * dt;
                self.degree_at(u, t) as f64
            })
            .sum();
        sum / n_samples as f64
    }

    /// Total contact time between nodes `u` and `v`.
    pub fn contact_duration(&self, u: usize, v: usize) -> f64 {
        self.edges
            .iter()
            .filter(|e| (e.src == u && e.dst == v) || (e.src == v && e.dst == u))
            .map(|e| e.duration())
            .sum()
    }

    /// Vector of gap durations between successive contacts of `u` and `v`.
    ///
    /// Contacts are sorted by start time; the gap is the time between the end
    /// of one contact and the start of the next.  Negative gaps (overlapping
    /// contacts) are excluded.
    pub fn inter_contact_times(&self, u: usize, v: usize) -> Vec<f64> {
        let mut contacts: Vec<(f64, f64)> = self
            .edges
            .iter()
            .filter(|e| (e.src == u && e.dst == v) || (e.src == v && e.dst == u))
            .map(|e| (e.start, e.end))
            .collect();
        contacts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        contacts
            .windows(2)
            .map(|w| w[1].0 - w[0].1)
            .filter(|&gap| gap > 0.0)
            .collect()
    }

    /// Local temporal clustering coefficient of node `u` at time `t`.
    ///
    /// Defined as the fraction of pairs among `u`'s neighbours that are
    /// themselves connected at time `t`.
    pub fn clustering_coefficient_at(&self, u: usize, t: f64) -> f64 {
        let neighbors: Vec<usize> = self
            .edges
            .iter()
            .filter(|e| e.is_active_at(t) && (e.src == u || e.dst == u))
            .map(|e| if e.src == u { e.dst } else { e.src })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let k = neighbors.len();
        if k < 2 {
            return 0.0;
        }

        let mut triangle_count = 0usize;
        for i in 0..k {
            for j in (i + 1)..k {
                let nb1 = neighbors[i];
                let nb2 = neighbors[j];
                let connected = self.edges.iter().any(|e| {
                    e.is_active_at(t)
                        && ((e.src == nb1 && e.dst == nb2) || (e.src == nb2 && e.dst == nb1))
                });
                if connected {
                    triangle_count += 1;
                }
            }
        }

        // Maximum possible pairs among k neighbours.
        let max_pairs = k * (k - 1) / 2;
        triangle_count as f64 / max_pairs as f64
    }

    /// Number of distinct nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of temporal edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Sum of all edge durations.
    pub fn total_interaction_time(&self) -> f64 {
        self.edges.iter().map(|e| e.duration()).sum()
    }

    /// Stream density: fraction of (node-pair × time) space that is active.
    ///
    /// Defined as `total_interaction_time / (|V|*(|V|-1)/2 * T)` for undirected streams.
    pub fn stream_density(&self) -> f64 {
        let n = self.nodes.len();
        if n < 2 {
            return 0.0;
        }
        let duration = self.time_end - self.time_start;
        if duration <= 0.0 {
            return 0.0;
        }
        let max_pairs = n * (n - 1) / 2;
        self.total_interaction_time() / (max_pairs as f64 * duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_stream() -> LinkStream {
        let mut ls = LinkStream::new(0.0, 10.0);
        ls.add_edge(0, 1, 0.0, 3.0);
        ls.add_edge(0, 2, 1.0, 4.0);
        ls.add_edge(1, 2, 2.0, 5.0);
        ls
    }

    #[test]
    fn test_contact_duration() {
        let ls = build_stream();
        assert!((ls.contact_duration(0, 1) - 3.0).abs() < 1e-9);
        assert!((ls.contact_duration(0, 2) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_degree_at() {
        let ls = build_stream();
        // At t=2.5, edges 0-1, 0-2, 1-2 are all active.
        assert_eq!(
            ls.degree_at(0, 2.5),
            2,
            "node 0 should have degree 2 at t=2.5"
        );
        assert_eq!(ls.degree_at(1, 2.5), 2);
    }

    #[test]
    fn test_inter_contact_times() {
        let mut ls = LinkStream::new(0.0, 20.0);
        ls.add_edge(0, 1, 0.0, 2.0);
        ls.add_edge(0, 1, 5.0, 7.0);
        ls.add_edge(0, 1, 12.0, 14.0);
        let ict = ls.inter_contact_times(0, 1);
        assert_eq!(ict.len(), 2, "ict={ict:?}");
        assert!((ict[0] - 3.0).abs() < 1e-9, "ict[0]={}", ict[0]);
        assert!((ict[1] - 5.0).abs() < 1e-9, "ict[1]={}", ict[1]);
    }

    #[test]
    fn test_clustering_coefficient() {
        let ls = build_stream();
        // At t=2.5 node 0 has neighbours {1,2} and they are connected → cc=1.0
        let cc = ls.clustering_coefficient_at(0, 2.5);
        assert!((cc - 1.0).abs() < 1e-9, "cc={cc}");
    }

    #[test]
    fn test_total_interaction_time() {
        let ls = build_stream();
        // 3 + 3 + 3 = 9
        assert!((ls.total_interaction_time() - 9.0).abs() < 1e-9);
    }
}
